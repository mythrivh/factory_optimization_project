import java.util.Scanner;

public class LinearSearch {
    @SuppressWarnings("ConvertToTryWithResources")
    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        // Input size of array
        System.out.print("Enter number of elements: ");
        int n = sc.nextInt();

        int[] arr = new int[n];

        // Input elements
        System.out.println("Enter the elements:");
        for (int i = 0; i < n; i++) {
            arr[i] = sc.nextInt();
        }

        // Input element to search
        System.out.print("Enter element to search: ");
        int key = sc.nextInt();

        // Linear Search logic
        int pos = -1;
        for (int i = 0; i < n; i++) {
            if (arr[i] == key) {
                pos = i; // store index
                break;
            }
        }

        // Output result
        if (pos == -1) {
            System.out.println("Element not found.");
        } else {
            // +1 to show position in human-friendly format
            System.out.println("Element found at position: " + (pos + 1));
        }

        sc.close();
    }
}
